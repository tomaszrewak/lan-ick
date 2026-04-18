"""Experiment 27: Layer combination comparison.

Test whether different layer selections meaningfully change performance.
Uses per-layer caches (already extracted for all 26 layers) merged on the fly.
Fast evaluation: 2-fold CV per combo, full 5-fold on baseline + best.
"""

import random
import time
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from src.cache import cached
from src.data import generate_synthetic_pairs
from src.classifier import (
    select_features_position_aware_topn, train_ovr, calibrate_greedy_f05, Metrics,
)
from src.pipeline import (
    calibration_split, _score_clean_error,
    N_PAIRS, MIN_WORDS, MAX_WORDS, DATA_SEED, DATA_VERSION,
    EXTRACT_VERSION, TOP_N, CALIB_RATIO,
)

FOLD_SEED = 42
DATA_CACHE_KEY = f"{DATA_VERSION}_n{N_PAIRS}"

# Layer combos to test
COMBOS = {
    "baseline":     [7, 13, 17, 22],
    "best_5":       [3, 7, 13, 17, 25],
    "skip_17":      [3, 7, 13, 25],
    "minimal_3":    [3, 7, 25],
}


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


def evaluate_layers(all_pairs, all_features, layers, folds, verbose=False):
    """Run K-fold CV for a given layer subset. Returns (mean_f05, std_f05, mean_p, mean_r)."""
    # Filter features to only requested layers
    layer_set = set(layers)
    filtered = []
    for feat in all_features:
        filtered.append({
            "clean": {
                "tokens": feat["clean"]["tokens"],
                "features": {l: feat["clean"]["features"].get(l, {}) for l in layer_set},
            },
            "error": {
                "tokens": feat["error"]["tokens"],
                "features": {l: feat["error"]["features"].get(l, {}) for l in layer_set},
            },
        })

    f05_scores, p_scores, r_scores = [], [], []

    for fold_num, (train_idx, test_idx) in enumerate(folds):
        fit_idx, calib_idx = calibration_split(train_idx, CALIB_RATIO, FOLD_SEED * 100 + fold_num)

        fit_pairs = [all_pairs[i] for i in fit_idx]
        fit_features = [filtered[i] for i in fit_idx]
        calib_pairs = [all_pairs[i] for i in calib_idx]
        calib_features = [filtered[i] for i in calib_idx]
        test_pairs = [all_pairs[i] for i in test_idx]
        test_features = [filtered[i] for i in test_idx]

        per_type_feats = select_features_position_aware_topn(
            fit_features, layers, fit_pairs, top_n=TOP_N, verbose=False,
        )
        classifier = train_ovr(fit_features, fit_pairs, layers, per_type_feats, verbose=False)
        types = list(classifier.models.keys())

        calib_err, calib_clean = _score_clean_error(calib_pairs, calib_features, classifier)
        test_err, test_clean = _score_clean_error(test_pairs, test_features, classifier)

        calibrate_greedy_f05(classifier, calib_err, calib_clean)
        cm = _combined_metrics(test_err, test_clean, classifier.thresholds, types)
        f05 = _f_beta(cm.precision, cm.recall, 0.5)
        f05_scores.append(f05)
        p_scores.append(cm.precision)
        r_scores.append(cm.recall)

        if verbose:
            print(f"    Fold {fold_num+1}: F0.5={f05:.1%}  P={cm.precision:.1%}  R={cm.recall:.1%}")

    return np.mean(f05_scores), np.std(f05_scores), np.mean(p_scores), np.mean(r_scores)


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

    # Collect all unique layers needed
    all_needed = set()
    for layers in COMBOS.values():
        all_needed.update(layers)
    print(f"Loading {len(all_needed)} unique layers: {sorted(all_needed)}")

    all_features = load_merged_features(all_pairs, all_needed)
    print(f"Loaded {len(all_pairs)} pairs with {len(all_needed)} layers\n")

    # Fast screening: 2-fold CV on each combo
    folds2 = kfold_split(len(all_pairs), 2, FOLD_SEED)
    print(f"{'='*60}")
    print("PHASE 1: Quick screening (2-fold CV)")
    print(f"{'='*60}")

    results = {}
    for name, layers in COMBOS.items():
        t0 = time.time()
        f05_m, f05_s, p_m, r_m = evaluate_layers(all_pairs, all_features, layers, folds2)
        elapsed = time.time() - t0
        results[name] = (f05_m, f05_s, p_m, r_m)
        tag = " <<<" if name == "baseline" else ""
        print(f"  {name:<15} {str(layers):<25} F0.5={f05_m:.1%}±{f05_s:.1%}  P={p_m:.1%}  R={r_m:.1%}  ({elapsed:.0f}s){tag}")

    # Rank by F0.5
    ranked = sorted(results.items(), key=lambda x: -x[1][0])
    print(f"\n  Ranking:")
    for i, (name, (f05, std, p, r)) in enumerate(ranked):
        delta = f05 - results["baseline"][0]
        print(f"    {i+1}. {name:<15} F0.5={f05:.1%}  Δ={delta:+.1%}")

    # Full 5-fold on baseline + best non-baseline
    best_name = ranked[0][0] if ranked[0][0] != "baseline" else (ranked[1][0] if len(ranked) > 1 else None)

    folds5 = kfold_split(len(all_pairs), 5, FOLD_SEED)
    print(f"\n{'='*60}")
    print("PHASE 2: Full 5-fold CV validation")
    print(f"{'='*60}")

    t0 = time.time()
    base_f05, base_std, base_p, base_r = evaluate_layers(
        all_pairs, all_features, COMBOS["baseline"], folds5, verbose=True,
    )
    print(f"  baseline {COMBOS['baseline']}: F0.5={base_f05:.1%}±{base_std:.1%}  P={base_p:.1%}  R={base_r:.1%}  ({time.time()-t0:.0f}s)")

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
