"""Experiment 22: Threshold calibration leakage fix + combined-precision framing.

Splits each training fold into fit / calibration subsets so per-type thresholds
are tuned on data disjoint from the final evaluation set. Reports F0.5 as the
primary metric alongside combined precision/recall.
"""

import random
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from src.data import ErrorType
from src.classifier import (
    select_features_position_aware_topn, train_ovr, predict_tokens_ovr, Metrics,
)
from src.pipeline import load_data, LAYERS, TOP_N, FP_BUDGET

K_FOLDS = 5
FOLD_SEED = 42


def _f_beta(p: float, r: float, beta: float) -> float:
    if p + r == 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def kfold_split(n: int, k: int, seed: int) -> list[tuple[list[int], list[int]]]:
    """Generate K train/test index splits."""
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


CALIB_RATIO = 0.2  # fraction of training fold held out for threshold calibration


def _score_pairs(pairs, features, classifier):
    """Score clean + error texts for each pair. Returns (error_scores, clean_scores)."""
    error_scores: list[dict[ErrorType, float]] = []
    clean_scores: list[dict[ErrorType, float]] = []
    for pf, pair in zip(features, pairs):
        for text_key, text, scores_list in [
            ("error", pair.error, error_scores),
            ("clean", pair.clean, clean_scores),
        ]:
            tpreds = predict_tokens_ovr(pf[text_key], classifier, text=text)
            scores = {
                et: max((tp.error_probs.get(et, 0.0) for tp in tpreds), default=0.0)
                for et in classifier.models
            }
            scores_list.append(scores)
    return error_scores, clean_scores


def evaluate_fold(all_pairs, all_features, train_idx, test_idx, fold_num):
    """Run full pipeline on one fold.

    Training fold is split further into fit_idx (feature selection + LR training)
    and calib_idx (per-type threshold selection under FP budget). Test fold is
    used only for the final reported metrics. This prevents the leakage where
    thresholds were previously fit on the same clean texts used for evaluation.
    """
    # Split training fold into fit + calibration (deterministic per fold)
    rng = random.Random(FOLD_SEED * 100 + fold_num)
    shuffled = list(train_idx)
    rng.shuffle(shuffled)
    n_calib = max(1, int(len(shuffled) * CALIB_RATIO))
    calib_idx = sorted(shuffled[:n_calib])
    fit_idx = sorted(shuffled[n_calib:])

    fit_pairs = [all_pairs[i] for i in fit_idx]
    fit_features = [all_features[i] for i in fit_idx]
    calib_pairs = [all_pairs[i] for i in calib_idx]
    calib_features = [all_features[i] for i in calib_idx]
    test_pairs = [all_pairs[i] for i in test_idx]
    test_features = [all_features[i] for i in test_idx]

    # Feature selection + training (on fit split only)
    per_type_feats = select_features_position_aware_topn(
        fit_features, LAYERS, fit_pairs, top_n=TOP_N,
    )
    classifier = train_ovr(fit_features, fit_pairs, LAYERS, per_type_feats)

    # Threshold calibration on held-out calibration split
    _, calib_clean_scores = _score_pairs(calib_pairs, calib_features, classifier)
    n_calib_clean = len(calib_clean_scores)
    max_fp_calib = int(n_calib_clean * FP_BUDGET)
    type_thresholds: dict[ErrorType, float] = {}
    for et in classifier.models:
        best_t = 1.0
        for t_int in range(50, 100):
            t = t_int / 100
            fp_count = sum(1 for cs in calib_clean_scores if cs.get(et, 0) >= t)
            if fp_count <= max_fp_calib:
                best_t = t
                break
        type_thresholds[et] = best_t

    # Final scoring on test set (unseen during fit OR calibration)
    error_scores, clean_scores = _score_pairs(test_pairs, test_features, classifier)
    n_clean = len(clean_scores)

    # Per-type detection rates
    per_type_det: dict[ErrorType, float] = {}
    per_type_fp: dict[ErrorType, float] = {}
    per_type_thresh: dict[ErrorType, float] = {}
    for et in ErrorType:
        if et not in type_thresholds:
            continue
        t = type_thresholds[et]
        type_idx = [i for i, p in enumerate(test_pairs) if p.error_type == et]
        n = len(type_idx)
        if n == 0:
            continue
        detected = sum(1 for i in type_idx if error_scores[i].get(et, 0) >= t)
        fp = sum(1 for cs in clean_scores if cs.get(et, 0) >= t)
        per_type_det[et] = detected / n
        per_type_fp[et] = fp / n_clean
        per_type_thresh[et] = t

    # Combined metrics
    tp = sum(1 for es in error_scores if any(
        es.get(et, 0) >= type_thresholds.get(et, 1.0) for et in classifier.models))
    fn = len(error_scores) - tp
    fp_total = sum(1 for cs in clean_scores if any(
        cs.get(et, 0) >= type_thresholds.get(et, 1.0) for et in classifier.models))
    tn = len(clean_scores) - fp_total
    cm = Metrics(tp=tp, fp=fp_total, tn=tn, fn=fn)

    return per_type_det, per_type_fp, per_type_thresh, cm


# --------------- Main ---------------

def main():
    # 1. Load data (cached)
    all_pairs, all_features = load_data()
    print(f"Loaded {len(all_pairs)} pairs")

    # 2. K-fold CV
    folds = kfold_split(len(all_pairs), K_FOLDS, FOLD_SEED)

    all_det: dict[ErrorType, list[float]] = {et: [] for et in ErrorType}
    all_fp: dict[ErrorType, list[float]] = {et: [] for et in ErrorType}
    all_thresh: dict[ErrorType, list[float]] = {et: [] for et in ErrorType}
    all_f1, all_p, all_r, all_fp_count = [], [], [], []
    all_f05 = []

    for fold_num, (train_idx, test_idx) in enumerate(folds):
        print(f"\n{'='*50}")
        print(f"FOLD {fold_num+1}/{K_FOLDS}  (train={len(train_idx)}, test={len(test_idx)})")
        print(f"{'='*50}")

        per_type_det, per_type_fp, per_type_thresh, cm = evaluate_fold(
            all_pairs, all_features, train_idx, test_idx, fold_num,
        )

        # Print fold results
        print(f"\n  {'Type':<15} {'Thresh':>6} {'Det':>5} {'FP':>5}")
        print(f"  {'-'*35}")
        for et in ErrorType:
            if et in per_type_det:
                print(f"  {et.value:<15} {per_type_thresh[et]:>6.2f} {per_type_det[et]:>4.0%} {per_type_fp[et]:>4.1%}")
                all_det[et].append(per_type_det[et])
                all_fp[et].append(per_type_fp[et])
                all_thresh[et].append(per_type_thresh[et])
        print(f"\n  Combined: F0.5={_f_beta(cm.precision, cm.recall, 0.5):.1%}  F1={cm.f1:.1%}  P={cm.precision:.1%}  R={cm.recall:.1%}  FP#={cm.fp}")
        all_f1.append(cm.f1)
        all_p.append(cm.precision)
        all_r.append(cm.recall)
        all_fp_count.append(cm.fp)
        all_f05.append(_f_beta(cm.precision, cm.recall, 0.5))

    # 3. Summary
    print(f"\n{'='*60}")
    print(f"  {K_FOLDS}-FOLD CV SUMMARY")
    print(f"{'='*60}")
    print(f"\n  {'Type':<15} {'Det mean':>8} {'Det std':>8} {'Thresh':>8} {'FP mean':>8}")
    print(f"  {'-'*50}")
    for et in ErrorType:
        if all_det[et]:
            det_arr = np.array(all_det[et])
            fp_arr = np.array(all_fp[et])
            thresh_arr = np.array(all_thresh[et])
            print(f"  {et.value:<15} {det_arr.mean():>7.1%} {det_arr.std():>7.1%} "
                  f"{thresh_arr.mean():>8.2f} {fp_arr.mean():>7.1%}")

    f1_arr = np.array(all_f1)
    p_arr = np.array(all_p)
    r_arr = np.array(all_r)
    fp_arr = np.array(all_fp_count)
    f05_arr = np.array(all_f05)
    print(f"\n  Combined F0.5: {f05_arr.mean():.1%} ± {f05_arr.std():.1%}  [PRIMARY]")
    print(f"  Precision:     {p_arr.mean():.1%} ± {p_arr.std():.1%}")
    print(f"  Recall:        {r_arr.mean():.1%} ± {r_arr.std():.1%}")
    print(f"  F1 (sanity):   {f1_arr.mean():.1%} ± {f1_arr.std():.1%}")
    print(f"  FP count:      {fp_arr.mean():.1f} ± {fp_arr.std():.1f}")
    print(f"\n  Per-fold F0.5: {[f'{f:.1%}' for f in all_f05]}")
    print(f"  Per-fold F1:   {[f'{f:.1%}' for f in all_f1]}")
    print(f"  Per-fold P:    {[f'{p:.1%}' for p in all_p]}")
    print(f"  Per-fold R:    {[f'{r:.1%}' for r in all_r]}")


if __name__ == "__main__":
    main()
