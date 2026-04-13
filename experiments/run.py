"""Experiment runner: OVR error detection with per-type features.

Load data → extract SAE features → select per-type features → train OVR → evaluate.
Best known config: 16k SAEs, layers [7,13,17,22], min_pair_ratio=0.3.
"""

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from src.cache import cached
from src.data import ErrorType, generate_synthetic_pairs, split_train_test
from src.model import extract_text_features
from src.classifier import (
    select_features_per_type, train_ovr, predict_tokens_ovr, Metrics,
)

# --------------- Parameters ---------------

LAYERS = [7, 13, 17, 22]
N_PAIRS = 600
MIN_WORDS = 8
MAX_WORDS = 20
DATA_SEED = 42
TRAIN_RATIO = 0.75
SPLIT_SEED = 42
MIN_PAIR_RATIO = 0.3

DATA_VERSION = "v5"
EXTRACT_VERSION = "v2"
EXTRACT_CACHE_KEY = f"{EXTRACT_VERSION}_{DATA_VERSION}_n{N_PAIRS}_layers={'_'.join(map(str, LAYERS))}_w16k"

THRESHOLDS = [0.5, 0.8, 0.9, 0.95]


# --------------- Main ---------------

def main():
    # 1. Load data
    all_pairs = cached(
        "synthetic_pairs", DATA_VERSION,
        lambda: generate_synthetic_pairs(N_PAIRS, MIN_WORDS, MAX_WORDS, DATA_SEED),
    )
    print(f"Loaded {len(all_pairs)} pairs")

    train_idx, test_idx = split_train_test(all_pairs, TRAIN_RATIO, SPLIT_SEED)
    train_pairs = [all_pairs[i] for i in train_idx]
    test_pairs = [all_pairs[i] for i in test_idx]
    print(f"Split: {len(train_pairs)} train, {len(test_pairs)} test")

    # 2. Extract features
    def extract_all():
        results = []
        for i, pair in enumerate(all_pairs):
            print(f"  [{i+1}/{len(all_pairs)}] {pair.error[:50]}...")
            clean_feats = extract_text_features(pair.clean, LAYERS)
            error_feats = extract_text_features(pair.error, LAYERS)
            results.append({"clean": clean_feats, "error": error_feats})
        return results

    all_features = cached("pair_features", EXTRACT_CACHE_KEY, extract_all)
    train_features = [all_features[i] for i in train_idx]
    test_features = [all_features[i] for i in test_idx]

    # 3. Select features per type
    per_type_feats = select_features_per_type(
        train_features, LAYERS, train_pairs, min_pair_ratio=MIN_PAIR_RATIO,
    )
    for et in ErrorType:
        feats = per_type_feats.get(et, {})
        total = sum(len(v) for v in feats.values())
        by_layer = ", ".join(f"L{l}:{len(feats.get(l, set()))}" for l in LAYERS)
        print(f"  {et.value}: {total} features ({by_layer})")

    # 4. Train OVR classifiers
    classifier = train_ovr(train_features, train_pairs, LAYERS, per_type_feats)

    # 5. Pre-compute scores on test set
    error_scores: list[dict[ErrorType, float]] = []
    clean_scores: list[dict[ErrorType, float]] = []

    for pf, pair in zip(test_features, test_pairs):
        for text_key, text, scores_list in [
            ("error", pair.error, error_scores),
            ("clean", pair.clean, clean_scores),
        ]:
            tpreds = predict_tokens_ovr(pf[text_key], classifier, text=text)
            scores = {}
            for et in classifier.models:
                scores[et] = max((tp.error_probs.get(et, 0.0) for tp in tpreds), default=0.0)
            scores_list.append(scores)

    # 6. Evaluate at each threshold
    for threshold in THRESHOLDS:
        print(f"\n--- threshold={threshold} ---")
        print(f"  {'Type':<15} {'Feats':>6} {'Det':>5} {'FP':>5}")
        print(f"  {'-'*35}")

        for et in ErrorType:
            n_feats = sum(len(v) for v in per_type_feats.get(et, {}).values())
            if et not in classifier.models:
                print(f"  {et.value:<15} {n_feats:>6}     —  0.0%")
                continue

            type_idx = [i for i, p in enumerate(test_pairs) if p.error_type == et]
            detected = sum(1 for i in type_idx if error_scores[i].get(et, 0) >= threshold)
            fp = sum(1 for cs in clean_scores if cs.get(et, 0) >= threshold)
            n = len(type_idx)
            det_pct = detected / n * 100 if n else 0
            fp_pct = fp / len(test_features) * 100
            print(f"  {et.value:<15} {n_feats:>6} {det_pct:>4.0f}% {fp_pct:>4.1f}%")

        tp = sum(1 for es in error_scores if any(
            es.get(et, 0) >= threshold for et in classifier.models))
        fn = len(error_scores) - tp
        fp_total = sum(1 for cs in clean_scores if any(
            cs.get(et, 0) >= threshold for et in classifier.models))
        tn = len(clean_scores) - fp_total
        cm = Metrics(tp=tp, fp=fp_total, tn=tn, fn=fn)
        print(f"  Combined: F1={cm.f1:.1%}  P={cm.precision:.1%}  R={cm.recall:.1%}  FP#={cm.fp}")


if __name__ == "__main__":
    main()
