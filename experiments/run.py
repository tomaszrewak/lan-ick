"""Experiment 29: Threshold cap impact.

Test how capping max_threshold at 0.90 (vs uncapped 1.00) affects combined
metrics and per-type detection. The hypothesis is that types currently at 0.99
(grammar, spelling, word_choice, word_order) are over-suppressed, and a lower
cap will trade some precision for much better recall.
"""

import random
import time
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from src.cache import cached
from src.data import ErrorType, generate_synthetic_pairs
from src.classifier import (
    select_features_position_aware_topn, train_ovr, calibrate_greedy_f05, Metrics,
)
from src.pipeline import (
    calibration_split, _score_clean_error, load_data,
    TOP_N, CALIB_RATIO, LAYERS,
)

FOLD_SEED = 42

# Threshold caps to sweep
MAX_THRESHOLDS = [1.00, 0.95, 0.90, 0.85, 0.80]


def _f_beta(p: float, r: float, beta: float) -> float:
    if p + r == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def _combined_metrics(err_scores, clean_scores, thresholds, types) -> Metrics:
    def fires(s):
        return any(s.get(e, 0.0) >= thresholds[e] for e in types)
    tp = sum(1 for es in err_scores if fires(es))
    fp = sum(1 for cs in clean_scores if fires(cs))
    return Metrics(tp=tp, fp=fp, tn=len(clean_scores) - fp, fn=len(err_scores) - tp)


def kfold_split(n: int, k: int, seed: int) -> list[tuple[list[int], list[int]]]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    fold_size = n // k
    folds = []
    for i in range(k):
        test_start = i * fold_size
        test_end = test_start + fold_size if i < k - 1 else n
        test_idx = sorted(indices[test_start:test_end])
        train_idx = sorted(set(indices) - set(test_idx))
        folds.append((train_idx, test_idx))
    return folds


def evaluate_cap(all_pairs, all_features, folds, max_threshold):
    """Run 5-fold CV with a given max_threshold cap. Returns metrics + per-type stats."""
    f05_scores, p_scores, r_scores = [], [], []
    per_type_detected = {}
    per_type_total = {}
    per_type_fp = {}
    all_thresholds = []

    for fold_num, (train_idx, test_idx) in enumerate(folds):
        fit_idx, calib_idx = calibration_split(train_idx, CALIB_RATIO, FOLD_SEED * 100 + fold_num)

        fit_pairs = [all_pairs[i] for i in fit_idx]
        fit_features = [all_features[i] for i in fit_idx]
        calib_pairs = [all_pairs[i] for i in calib_idx]
        calib_features = [all_features[i] for i in calib_idx]
        test_pairs = [all_pairs[i] for i in test_idx]
        test_features = [all_features[i] for i in test_idx]

        per_type_feats = select_features_position_aware_topn(
            fit_features, LAYERS, fit_pairs, top_n=TOP_N, verbose=False,
        )
        classifier = train_ovr(fit_features, fit_pairs, LAYERS, per_type_feats, verbose=False)
        types = list(classifier.models.keys())

        calib_err, calib_clean = _score_clean_error(calib_pairs, calib_features, classifier)
        test_err, test_clean = _score_clean_error(test_pairs, test_features, classifier)

        calibrate_greedy_f05(classifier, calib_err, calib_clean, max_threshold=max_threshold)
        all_thresholds.append(dict(classifier.thresholds))

        cm = _combined_metrics(test_err, test_clean, classifier.thresholds, types)
        f05 = _f_beta(cm.precision, cm.recall, 0.5)
        f05_scores.append(f05)
        p_scores.append(cm.precision)
        r_scores.append(cm.recall)

        # Per-type detection
        test_err_pairs = [all_pairs[i] for i in test_idx]
        for et in types:
            et_total = sum(1 for p in test_err_pairs if p.error_type == et)
            et_detected = sum(
                1 for es, p in zip(test_err, test_err_pairs)
                if p.error_type == et and es.get(et, 0.0) >= classifier.thresholds[et]
            )
            et_fp = sum(1 for cs in test_clean if cs.get(et, 0.0) >= classifier.thresholds[et])
            per_type_detected.setdefault(et, []).append(et_detected)
            per_type_total.setdefault(et, []).append(et_total)
            per_type_fp.setdefault(et, []).append(et_fp)

    # Aggregate
    avg_thresh = {}
    for et in all_thresholds[0]:
        avg_thresh[et] = np.mean([t[et] for t in all_thresholds])

    per_type_stats = {}
    for et in per_type_detected:
        total = sum(per_type_total[et])
        detected = sum(per_type_detected[et])
        fp = sum(per_type_fp[et])
        per_type_stats[et] = {"det": detected / total if total else 0, "detected": detected, "total": total, "fp": fp}

    return {
        "f05": np.mean(f05_scores), "f05_std": np.std(f05_scores),
        "p": np.mean(p_scores), "r": np.mean(r_scores),
        "thresholds": avg_thresh, "per_type": per_type_stats,
    }


def main():
    all_pairs, all_features = load_data()
    print(f"Loaded {len(all_pairs)} pairs\n")

    folds = kfold_split(len(all_pairs), 5, FOLD_SEED)

    print(f"{'='*70}")
    print("Threshold cap sweep (5-fold CV)")
    print(f"  Caps: {MAX_THRESHOLDS}")
    print(f"{'='*70}\n")

    results = {}
    for cap in MAX_THRESHOLDS:
        t0 = time.time()
        r = evaluate_cap(all_pairs, all_features, folds, cap)
        elapsed = time.time() - t0
        results[cap] = r

        tag = " <<< baseline" if cap == 1.00 else ""
        print(f"  cap={cap:.2f}  F0.5={r['f05']:.1%}±{r['f05_std']:.1%}  P={r['p']:.1%}  R={r['r']:.1%}  ({elapsed:.0f}s){tag}")

        # Per-type breakdown
        for et in sorted(r['per_type'], key=lambda e: e.value):
            s = r['per_type'][et]
            print(f"    {et.value:<12} det={s['det']:.0%} ({s['detected']}/{s['total']})  FP={s['fp']}  thresh={r['thresholds'].get(et, 0):.2f}")
        print()

    # Summary
    print(f"\n  Ranking by F0.5:")
    baseline_f05 = results[1.00]['f05']
    ranked = sorted(results.items(), key=lambda x: -x[1]['f05'])
    for i, (cap, r) in enumerate(ranked):
        delta = r['f05'] - baseline_f05
        print(f"    {i+1}. cap={cap:.2f}  F0.5={r['f05']:.1%}±{r['f05_std']:.1%}  P={r['p']:.1%}  R={r['r']:.1%}  Δ={delta:+.1%}")


if __name__ == "__main__":
    main()

    if best_name:
        t0 = time.time()
        best_f05, best_std, best_p, best_r = evaluate_layers(
            all_pairs, all_features, COMBOS[best_name], folds5, verbose=True,
        )
        print(f"  {best_name} {COMBOS[best_name]}: F0.5={best_f05:.1%}±{best_std:.1%}  P={best_p:.1%}  R={best_r:.1%}  ({time.time()-t0:.0f}s)")
        delta = best_f05 - base_f05
        print(f"\n  Δ(best vs baseline) = {delta:+.1%}")


if __name__ == "__main__":
    main()
