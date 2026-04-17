"""K-fold CV runner over the best-known pipeline.

Each fold: split train into fit+calibration (80/20), train OVR on fit,
calibrate per-type thresholds by greedy-F0.5 coordinate descent on calibration,
evaluate on the held-out test fold. Reports combined F0.5 (primary), precision,
recall, F1.
"""

import random
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from src.classifier import (
    select_features_position_aware_topn, train_ovr, calibrate_greedy_f05, Metrics,
)
from src.pipeline import (
    load_data, calibration_split, _score_clean_error,
    LAYERS, TOP_N, CALIB_RATIO,
)

K_FOLDS = 5
FOLD_SEED = 42


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


def evaluate_fold(all_pairs, all_features, train_idx, test_idx, fold_num):
    fit_idx, calib_idx = calibration_split(train_idx, CALIB_RATIO, FOLD_SEED * 100 + fold_num)

    fit_pairs = [all_pairs[i] for i in fit_idx]
    fit_features = [all_features[i] for i in fit_idx]
    calib_pairs = [all_pairs[i] for i in calib_idx]
    calib_features = [all_features[i] for i in calib_idx]
    test_pairs = [all_pairs[i] for i in test_idx]
    test_features = [all_features[i] for i in test_idx]

    per_type_feats = select_features_position_aware_topn(
        fit_features, LAYERS, fit_pairs, top_n=TOP_N,
    )
    classifier = train_ovr(fit_features, fit_pairs, LAYERS, per_type_feats)
    types = list(classifier.models.keys())

    calib_err, calib_clean = _score_clean_error(calib_pairs, calib_features, classifier)
    test_err, test_clean = _score_clean_error(test_pairs, test_features, classifier)

    calibrate_greedy_f05(classifier, calib_err, calib_clean)
    return _combined_metrics(test_err, test_clean, classifier.thresholds, types), classifier.thresholds, types


def main():
    all_pairs, all_features = load_data()
    print(f"Loaded {len(all_pairs)} pairs")

    folds = kfold_split(len(all_pairs), K_FOLDS, FOLD_SEED)
    metrics = {"f05": [], "f1": [], "p": [], "r": [], "fp": []}
    thresholds_per_fold: list[dict] = []

    for fold_num, (train_idx, test_idx) in enumerate(folds):
        print(f"\n{'='*60}\nFOLD {fold_num+1}/{K_FOLDS}\n{'='*60}")
        cm, thresholds, types = evaluate_fold(all_pairs, all_features, train_idx, test_idx, fold_num)
        thresholds_per_fold.append(thresholds)

        f05 = _f_beta(cm.precision, cm.recall, 0.5)
        metrics["f05"].append(f05)
        metrics["f1"].append(cm.f1)
        metrics["p"].append(cm.precision)
        metrics["r"].append(cm.recall)
        metrics["fp"].append(cm.fp)

        print(f"  F0.5={f05:.1%}  P={cm.precision:.1%}  R={cm.recall:.1%}  F1={cm.f1:.1%}  FP#={cm.fp}")
        print(f"  thresholds: " + ", ".join(f"{e.value}={thresholds[e]:.2f}" for e in types))

    print(f"\n{'='*60}\n  {K_FOLDS}-FOLD CV SUMMARY\n{'='*60}")
    def ms(key):
        a = np.array(metrics[key])
        return a.mean(), a.std()
    f05_m, f05_s = ms("f05")
    p_m, p_s = ms("p")
    r_m, r_s = ms("r")
    f1_m, f1_s = ms("f1")
    fp_m, fp_s = ms("fp")
    print(f"  F0.5 = {f05_m:.1%} ± {f05_s:.1%}")
    print(f"  P    = {p_m:.1%} ± {p_s:.1%}")
    print(f"  R    = {r_m:.1%} ± {r_s:.1%}")
    print(f"  F1   = {f1_m:.1%} ± {f1_s:.1%}")
    print(f"  FP#  = {fp_m:.0f} ± {fp_s:.0f}")

    print(f"\n  Greedy F0.5 thresholds (mean across folds):")
    type_names = sorted({e for fd in thresholds_per_fold for e in fd}, key=lambda e: e.value)
    for et in type_names:
        vals = [fd[et] for fd in thresholds_per_fold]
        print(f"    {et.value:<12} mean={np.mean(vals):.3f}  folds={[f'{v:.2f}' for v in vals]}")


if __name__ == "__main__":
    main()
