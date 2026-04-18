"""Experiment 28: Reduce per-type feature count for contaminated types.

Test whether using fewer features for grammar/word_order/word_choice improves
combined F0.5. These types have token-keyed candidate pools (Exp 25), so
more features = more noise. Sweep contaminated top_N while keeping clean types at 100.
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
    calibration_split, _score_clean_error,
    N_PAIRS, MIN_WORDS, MAX_WORDS, DATA_SEED, DATA_VERSION,
    EXTRACT_VERSION, LAYERS, CALIB_RATIO,
)

FOLD_SEED = 42
DATA_CACHE_KEY = f"{DATA_VERSION}_n{N_PAIRS}"

# Contaminated types (high token-keyed feature ratio from Exp 25)
CONTAMINATED = {ErrorType.GRAMMAR, ErrorType.WORD_ORDER, ErrorType.WORD_CHOICE}
CLEAN_TYPES = {ErrorType.SPELLING, ErrorType.EXTRA_WORD, ErrorType.WTF}
CLEAN_TOP_N = 100

# Sweep values for contaminated types
SWEEP_VALUES = [1, 2, 5, 10, 20, 100]  # find the floor


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


def make_top_n_dict(contaminated_n: int) -> dict[ErrorType, int]:
    """Build per-type top_n: contaminated types get contaminated_n, clean types get CLEAN_TOP_N."""
    d = {}
    for et in ErrorType:
        d[et] = contaminated_n if et in CONTAMINATED else CLEAN_TOP_N
    return d


def evaluate_top_n(all_pairs, all_features, top_n_dict, folds):
    """Run K-fold CV with per-type top_n. Returns (mean_f05, std_f05, mean_p, mean_r, avg_thresh, per_type_stats)."""
    f05_scores, p_scores, r_scores = [], [], []
    all_thresholds = []
    # Per-type: track detection count and total per error type, plus FP count on clean
    per_type_detected = {}  # {et: [count_per_fold]}
    per_type_total = {}     # {et: [total_per_fold]}
    per_type_fp = {}        # {et: [fp_count_per_fold]}

    for fold_num, (train_idx, test_idx) in enumerate(folds):
        fit_idx, calib_idx = calibration_split(train_idx, CALIB_RATIO, FOLD_SEED * 100 + fold_num)

        fit_pairs = [all_pairs[i] for i in fit_idx]
        fit_features = [all_features[i] for i in fit_idx]
        calib_pairs = [all_pairs[i] for i in calib_idx]
        calib_features = [all_features[i] for i in calib_idx]
        test_pairs = [all_pairs[i] for i in test_idx]
        test_features = [all_features[i] for i in test_idx]

        per_type_feats = select_features_position_aware_topn(
            fit_features, LAYERS, fit_pairs, top_n=top_n_dict, verbose=False,
        )
        classifier = train_ovr(fit_features, fit_pairs, LAYERS, per_type_feats, verbose=False)
        types = list(classifier.models.keys())

        calib_err, calib_clean = _score_clean_error(calib_pairs, calib_features, classifier)
        test_err, test_clean = _score_clean_error(test_pairs, test_features, classifier)

        calibrate_greedy_f05(classifier, calib_err, calib_clean)
        all_thresholds.append(dict(classifier.thresholds))

        cm = _combined_metrics(test_err, test_clean, classifier.thresholds, types)
        f05 = _f_beta(cm.precision, cm.recall, 0.5)
        f05_scores.append(f05)
        p_scores.append(cm.precision)
        r_scores.append(cm.recall)

        # Per-type detection: for each error sentence, check if its own type fires
        test_err_pairs = [all_pairs[i] for i in test_idx]
        for et in types:
            et_pairs_mask = [p.error_type == et for p in test_err_pairs]
            et_total = sum(et_pairs_mask)
            et_detected = sum(
                1 for es, is_et in zip(test_err, et_pairs_mask)
                if is_et and es.get(et, 0.0) >= classifier.thresholds[et]
            )
            per_type_detected.setdefault(et, []).append(et_detected)
            per_type_total.setdefault(et, []).append(et_total)

            # FP: clean sentences where this type fires
            et_fp = sum(
                1 for cs in test_clean
                if cs.get(et, 0.0) >= classifier.thresholds[et]
            )
            per_type_fp.setdefault(et, []).append(et_fp)

    # Averages
    avg_thresh = {}
    for et in all_thresholds[0]:
        avg_thresh[et] = np.mean([t[et] for t in all_thresholds])

    per_type_stats = {}
    for et in per_type_detected:
        total = sum(per_type_total[et])
        detected = sum(per_type_detected[et])
        fp = sum(per_type_fp[et])
        det_rate = detected / total if total else 0.0
        per_type_stats[et] = {"det": det_rate, "detected": detected, "total": total, "fp": fp}

    return np.mean(f05_scores), np.std(f05_scores), np.mean(p_scores), np.mean(r_scores), avg_thresh, per_type_stats


def load_merged_features(all_pairs, needed_layers):
    """Load per-layer caches and merge into a single feature list."""
    per_layer = {}
    for layer in sorted(needed_layers):
        key = f"{EXTRACT_VERSION}_{DATA_VERSION}_n{N_PAIRS}_layer{layer}_w16k"
        per_layer[layer] = cached(f"pair_features_L{layer}", key, lambda: None)
        if per_layer[layer] is None:
            raise RuntimeError(f"Cache missing for layer {layer} — run extraction first")

    first_layer = sorted(needed_layers)[0]
    all_features = []
    for i in range(len(all_pairs)):
        merged = {
            "clean": {
                "tokens": per_layer[first_layer][i]["clean"]["tokens"],
                "features": {},
            },
            "error": {
                "tokens": per_layer[first_layer][i]["error"]["tokens"],
                "features": {},
            },
        }
        for layer in needed_layers:
            merged["clean"]["features"][layer] = per_layer[layer][i]["clean"]["features"].get(layer, {})
            merged["error"]["features"][layer] = per_layer[layer][i]["error"]["features"].get(layer, {})
        all_features.append(merged)

    del per_layer
    import gc; gc.collect()
    return all_features


def main():
    all_pairs = cached(
        "synthetic_pairs", DATA_CACHE_KEY,
        lambda: generate_synthetic_pairs(N_PAIRS, MIN_WORDS, MAX_WORDS, DATA_SEED),
    )

    print(f"Loading layers {LAYERS}")
    all_features = load_merged_features(all_pairs, LAYERS)
    print(f"Loaded {len(all_pairs)} pairs\n")

    folds = kfold_split(len(all_pairs), 5, FOLD_SEED)

    print(f"{'='*70}")
    print("Per-type top_N sweep (5-fold CV)")
    print(f"  Clean types (spelling, extra_word, wtf): fixed at {CLEAN_TOP_N}")
    print(f"  Contaminated types (grammar, word_order, word_choice): sweep {SWEEP_VALUES}")
    print(f"{'='*70}\n")

    results = {}
    for contam_n in SWEEP_VALUES:
        top_n_dict = make_top_n_dict(contam_n)
        t0 = time.time()
        f05_m, f05_s, p_m, r_m, avg_thresh, per_type = evaluate_top_n(
            all_pairs, all_features, top_n_dict, folds,
        )
        elapsed = time.time() - t0
        results[contam_n] = (f05_m, f05_s, p_m, r_m, avg_thresh, per_type)

        tag = " <<< baseline" if contam_n == 100 else ""
        print(f"  contam_N={contam_n:<3}  F0.5={f05_m:.1%}±{f05_s:.1%}  P={p_m:.1%}  R={r_m:.1%}  ({elapsed:.0f}s){tag}")

        # Per-type breakdown
        for et in sorted(per_type, key=lambda e: e.value):
            s = per_type[et]
            print(f"    {et.value:<12} det={s['det']:.0%} ({s['detected']}/{s['total']})  FP={s['fp']}  thresh={avg_thresh.get(et, 0):.2f}")
        print()

    # Summary ranking
    print(f"\n  Ranking by F0.5:")
    ranked = sorted(results.items(), key=lambda x: -x[1][0])
    baseline_f05 = results[100][0]
    for i, (n, (f05, std, p, r, _, _pt)) in enumerate(ranked):
        delta = f05 - baseline_f05
        print(f"    {i+1}. contam_N={n:<3} F0.5={f05:.1%}±{std:.1%}  Δ={delta:+.1%}")

    # Diagnostic: show top features per type at N=1
    print(f"\n{'='*70}")
    print("Diagnostic: top-1 feature per type (full dataset)")
    print(f"{'='*70}")
    top1_dict = {et: 1 for et in ErrorType}
    feats_1 = select_features_position_aware_topn(
        all_features, LAYERS, all_pairs, top_n=top1_dict, verbose=True,
    )
    for et, layer_feats in sorted(feats_1.items(), key=lambda x: x[0].value):
        for layer, fids in sorted(layer_feats.items()):
            for fid in sorted(fids):
                print(f"  {et.value:<12} → layer {layer}, feature {fid}")


if __name__ == "__main__":
    main()
